namespace GEN.ORT.YYY
{
  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
  //
  //                    Source Code Generated by
  //                           CA Gen 8.6
  //
  //    Copyright (c) 2024 CA Technologies. All rights reserved.
  //
  //    Name: MYY10131_PARENT_UPDATE           Date: 2024/01/09
  //    Target OS:   CLR                       Time: 13:42:01
  //    Target DBMS: ODBC/ADO.NET              User: AliAl
  //    Access Method: <NONE>         
  //
  //    Generation options:
  //    Debug trace option not selected
  //    Data modeling constraint enforcement not selected
  //    Optimized import view initialization not selected
  //    Enforce default values with DBMS not selected
  //    Init unspecified optional fields to NULL not selected
  //
  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
  // using Statements
  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
  using System;
  using com.ca.gen.vwrt;
  using com.ca.gen.vwrt.types;
  using com.ca.gen.vwrt.vdf;
  using com.ca.gen.csu.exception;
  
  using com.ca.gen.abrt;
  using com.ca.gen.abrt.functions;
  using com.ca.gen.abrt.cascade;
  using com.ca.gen.abrt.manager;
  using com.ca.gen.abrt.trace;
  using com.ca.gen.exits.common;
  using com.ca.gen.odc;
  using System.Data;
  using System.Collections;
  
  public class MYY10131 : ABBase
  {
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // IMPORT VIEW CLASS VARIABLE
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    MYY10131_IA WIa;
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // EXPORT VIEW CLASS VARIABLE
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    MYY10131_OA WOa;
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // LOCAL VIEW CLASS VARIABLE
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    MYY10131_LA WLa;
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // ACTION BLOCK IMPORT/EXPORT VIEWS CLASS VARIABLES
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    GEN.ORT.YYY.CYYY0131_IA Cyyy0131Ia;
    GEN.ORT.YYY.CYYY0131_OA Cyyy0131Oa;
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // MISC DECLARATIONS AND PROTOTYPES 
    //    FOLLOW AS NEEDED:             
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    bool func_0022020333_esc_flag;
    //       +->   MYY10131_PARENT_UPDATE            01/09/2024  13:42
    //       !       IMPORTS:
    //       !         Work View imp_reference iyy1_server_data (Transient,
    //       !                     Mandatory, Import only)
    //       !           userid
    //       !           reference_id
    //       !         Entity View imp iyy1_parent (Transient, Mandatory,
    //       !                     Import only)
    //       !           pinstance_id
    //       !           preference_id
    //       !           pkey_attr_text
    //       !           psearch_attr_text
    //       !           pother_attr_text
    //       !           ptype_tkey_attr_text
    //       !       EXPORTS:
    //       !         Entity View exp iyy1_parent (Transient, Export only)
    //       !           preference_id
    //       !         Work View exp_error iyy1_component (Transient, Export
    //       !                     only)
    //       !           severity_code
    //       !           rollback_indicator
    //       !           origin_servid
    //       !           context_string
    //       !           return_code
    //       !           reason_code
    //       !           checksum
    //       !       LOCALS:
    //       !         Entity View loc_imp parent
    //       !           pinstance_id
    //       !           preference_id
    //       !           pkey_attr_text
    //       !           psearch_attr_text
    //       !           pother_attr_text
    //       !           ptype_tkey_attr_text
    //       !         Entity View loc_exp parent
    //       !           preference_id
    //       !
    //       !     PROCEDURE STATEMENTS
    //       !
    //     1 !  NOTE: 
    //     1 !  See the description for the purpose
    //     1 !  
    //     2 !  NOTE: 
    //     2 !  RELEASE HISTORY
    //     2 !  01_00 23-02-1998 New release
    //     2 !  
    //     3 !  NOTE: 
    //     3 !  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //     3 !  !!!!!!!!!!!!
    //     3 !  SET <loc imp*> TO <imp*>
    //     3 !  
    //     4 !  SET loc_imp parent pinstance_id TO imp iyy1_parent
    //     4 !              pinstance_id 
    //     5 !  SET loc_imp parent preference_id TO imp iyy1_parent
    //     5 !              preference_id 
    //     6 !  SET loc_imp parent pkey_attr_text TO imp iyy1_parent
    //     6 !              pkey_attr_text 
    //     7 !  SET loc_imp parent psearch_attr_text TO imp iyy1_parent
    //     7 !              psearch_attr_text 
    //     8 !  SET loc_imp parent pother_attr_text TO imp iyy1_parent
    //     8 !              pother_attr_text 
    //     9 !  SET loc_imp parent ptype_tkey_attr_text TO imp iyy1_parent
    //     9 !              ptype_tkey_attr_text 
    //    10 !   
    //    11 !  NOTE: 
    //    11 !  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //    11 !  !!!!!!!!!!!!
    //    11 !  USE <implementation ab>
    //    11 !  
    //    12 !  USE cyyy0131_parent_update
    //    12 !     WHICH IMPORTS: Entity View loc_imp parent TO Entity View
    //    12 !              imp parent
    //    12 !                    Work View imp_reference iyy1_server_data TO
    //    12 !              Work View imp_reference iyy1_server_data
    //    12 !     WHICH EXPORTS: Entity View loc_exp parent FROM Entity View
    //    12 !              exp parent
    //    12 !                    Work View exp_error iyy1_component FROM Work
    //    12 !              View exp_error iyy1_component
    //    13 !   
    //    14 !  NOTE: 
    //    14 !  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //    14 !  !!!!!!!!!!!!
    //    14 !  SET <exp*> TO <loc exp*>
    //    14 !  
    //    15 !  SET exp iyy1_parent preference_id TO loc_exp parent
    //    15 !              preference_id 
    //       +---
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    //  CONSTRUCTOR FOR THE CLASS       
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    public MYY10131(  )
    {
      IefCGenRlse = "CA Gen 8.6";
      IsCopyright = "Copyright (c) 2024 CA Technologies. All rights reserved.";
      IefCGenDate = "2024/01/09";
      IefCGenTime = "13:42:01";
      IefCGenEncy = "9.2.A6";
      IefCGenUserId = "AliAl";
      IefCGenModel = "N8I_ORT_YYY_0112_TEMPLATE";
      IefCGenSubset = "ALL";
      IefCGenName = "MYY10131_PARENT_UPDATE";
      NestingLevel = 0;
      ValChkDeadlockTimeout = false;
      ValChkDBError = false;
    }
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // ACTION BLOCK FUNCTION DECLARATIONS 
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    
    public void Execute( Object in_runtime_parm1, 
    	IRuntimePStepContext in_runtime_parm2, 
    	GlobData in_globdata, 
    	MYY10131_IA import_view, 
    	MYY10131_OA export_view )
    {
      IefRuntimeParm1 = in_runtime_parm1;
      IefRuntimeParm2 = in_runtime_parm2;
      Globdata = in_globdata;
      WIa = import_view;
      WOa = export_view;
      _Execute();
    }
    
    private void _Execute()
    {
      
      f_22020333_localAlloc( "MYY10131_PARENT_UPDATE" );
      if ( Globdata.GetErrorData().GetLastStatus() == ErrorData.LastStatusIefAllocationError )
      	return;
      
      ++(NestingLevel);
      try {
        f_22020333_init(  );
        f_22020333(  );
      } catch( Exception e ) {
        if ( ((Globdata.GetErrorData().GetStatus() == ErrorData.StatusNone) && (Globdata.GetErrorData().GetErrorEncountered() == 
          ErrorData.ErrorEncounteredNoErrorFound)) && (Globdata.GetErrorData().GetViewOverflow() == 
          ErrorData.ErrorEncounteredNoErrorFound) )
        {
          Globdata.GetErrorData().SetStatus( ErrorData.LastStatusFatalError );
          Globdata.GetErrorData().SetLastStatus( ErrorData.LastStatusUnexpectedExceptionError );
          Globdata.GetErrorData().SetErrorEncountered( ErrorData.ErrorEncounteredErrorFound );
          Globdata.GetErrorData(  ).SetErrorMessage( e );
        }
      }
      --(NestingLevel);
    }
    public void f_22020333(  )
    {
      func_0022020333_esc_flag = false;
      Globdata.GetStateData().SetCurrentABId( "0022020333" );
      Globdata.GetStateData().SetCurrentABName( "MYY10131_PARENT_UPDATE" );
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    See the description for the purpose                             
      //                                                                    
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    RELEASE HISTORY                                                 
      //    01_00 23-02-1998 New release                                    
      //                                                                    
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  
      //    !!!!!!!!!!!!                                                    
      //    SET <loc imp*> TO <imp*>                                        
      //                                                                    
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      Globdata.GetStateData().SetLastStatementNumber( "0000000004" );
      WLa.LocImpParentPinstanceId = TimestampAttr.ValueOf(WIa.ImpIyy1ParentPinstanceId);
      Globdata.GetStateData().SetLastStatementNumber( "0000000005" );
      WLa.LocImpParentPreferenceId = TimestampAttr.ValueOf(WIa.ImpIyy1ParentPreferenceId);
      Globdata.GetStateData().SetLastStatementNumber( "0000000006" );
      WLa.LocImpParentPkeyAttrText = FixedStringAttr.ValueOf(WIa.ImpIyy1ParentPkeyAttrText, 5);
      Globdata.GetStateData().SetLastStatementNumber( "0000000007" );
      WLa.LocImpParentPsearchAttrText = FixedStringAttr.ValueOf(WIa.ImpIyy1ParentPsearchAttrText, 25);
      Globdata.GetStateData().SetLastStatementNumber( "0000000008" );
      WLa.LocImpParentPotherAttrText = FixedStringAttr.ValueOf(WIa.ImpIyy1ParentPotherAttrText, 25);
      Globdata.GetStateData().SetLastStatementNumber( "0000000009" );
      WLa.LocImpParentPtypeTkeyAttrText = FixedStringAttr.ValueOf(WIa.ImpIyy1ParentPtypeTkeyAttrText, 4);
      
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  
      //    !!!!!!!!!!!!                                                    
      //    USE <implementation ab>                                         
      //                                                                    
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      Globdata.GetStateData().SetLastStatementNumber( "0000000012" );
      
      Cyyy0131Ia = (GEN.ORT.YYY.CYYY0131_IA)(IefRuntimeParm2.GetInstance( typeof(GEN.ORT.YYY.CYYY0131).Assembly,
      	"GEN.ORT.YYY.CYYY0131_IA" ));
      Cyyy0131Oa = (GEN.ORT.YYY.CYYY0131_OA)(IefRuntimeParm2.GetInstance( typeof(GEN.ORT.YYY.CYYY0131).Assembly,
      	"GEN.ORT.YYY.CYYY0131_OA" ));
      Cyyy0131Ia.ImpReferenceIyy1ServerDataUserid = FixedStringAttr.ValueOf(WIa.ImpReferenceIyy1ServerDataUserid, 8);
      Cyyy0131Ia.ImpReferenceIyy1ServerDataReferenceId = TimestampAttr.ValueOf(WIa.ImpReferenceIyy1ServerDataReferenceId);
      Cyyy0131Ia.ImpParentPinstanceId = TimestampAttr.ValueOf(WLa.LocImpParentPinstanceId);
      Cyyy0131Ia.ImpParentPreferenceId = TimestampAttr.ValueOf(WLa.LocImpParentPreferenceId);
      Cyyy0131Ia.ImpParentPkeyAttrText = FixedStringAttr.ValueOf(WLa.LocImpParentPkeyAttrText, 5);
      Cyyy0131Ia.ImpParentPsearchAttrText = FixedStringAttr.ValueOf(WLa.LocImpParentPsearchAttrText, 25);
      Cyyy0131Ia.ImpParentPotherAttrText = FixedStringAttr.ValueOf(WLa.LocImpParentPotherAttrText, 25);
      Cyyy0131Ia.ImpParentPtypeTkeyAttrText = FixedStringAttr.ValueOf(WLa.LocImpParentPtypeTkeyAttrText, 4);
      Globdata.GetErrorData().SetErrorEncountered( ErrorData.ErrorEncounteredNoErrorFound );
      IefRuntimeParm2.UseActionBlock( typeof(GEN.ORT.YYY.CYYY0131).Assembly,
      	"GEN.ORT.YYY.CYYY0131",
      	"Execute",
      	Cyyy0131Ia,
      	Cyyy0131Oa );
      if ( ((Globdata.GetErrorData().GetStatus() != ErrorData.StatusNone) || (Globdata.GetErrorData().GetErrorEncountered() != 
        ErrorData.ErrorEncounteredNoErrorFound)) || (Globdata.GetErrorData().GetViewOverflow() != 
        ErrorData.ErrorEncounteredNoErrorFound) )
      {
        throw new ABException();
      }
      Globdata.GetStateData().SetCurrentABId( "0022020333" );
      Globdata.GetStateData().SetCurrentABName( "MYY10131_PARENT_UPDATE" );
      Globdata.GetStateData().SetLastStatementNumber( "0000000012" );
      WOa.ExpErrorIyy1ComponentSeverityCode = FixedStringAttr.ValueOf(Cyyy0131Oa.ExpErrorIyy1ComponentSeverityCode, 1);
      WOa.ExpErrorIyy1ComponentRollbackIndicator = FixedStringAttr.ValueOf(Cyyy0131Oa.ExpErrorIyy1ComponentRollbackIndicator, 1);
      WOa.ExpErrorIyy1ComponentOriginServid = DoubleAttr.ValueOf(Cyyy0131Oa.ExpErrorIyy1ComponentOriginServid);
      WOa.ExpErrorIyy1ComponentContextString = StringAttr.ValueOf(Cyyy0131Oa.ExpErrorIyy1ComponentContextString, 512);
      WOa.ExpErrorIyy1ComponentReturnCode = IntAttr.ValueOf(Cyyy0131Oa.ExpErrorIyy1ComponentReturnCode);
      WOa.ExpErrorIyy1ComponentReasonCode = IntAttr.ValueOf(Cyyy0131Oa.ExpErrorIyy1ComponentReasonCode);
      WOa.ExpErrorIyy1ComponentChecksum = FixedStringAttr.ValueOf(Cyyy0131Oa.ExpErrorIyy1ComponentChecksum, 15);
      WLa.LocExpParentPreferenceId = TimestampAttr.ValueOf(Cyyy0131Oa.ExpParentPreferenceId);
      Cyyy0131Ia.FreeInstance(  );
      Cyyy0131Ia = null;
      Cyyy0131Oa.FreeInstance(  );
      Cyyy0131Oa = null;
      
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  
      //    !!!!!!!!!!!!                                                    
      //    SET <exp*> TO <loc exp*>                                        
      //                                                                    
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      Globdata.GetStateData().SetLastStatementNumber( "0000000015" );
      WOa.ExpIyy1ParentPreferenceId = TimestampAttr.ValueOf(WLa.LocExpParentPreferenceId);
      return;
    }
    
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // SUBORDINATE FUNCTIONS
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // INITIALIZATION UTILITY FUNCTIONS 
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    
    public void f_22020333_localAlloc( String abname )
    {
      // Request localview allocation 
      WLa = (GEN.ORT.YYY.MYY10131_LA)(IefRuntimeParm2.GetInstance( GetType().Assembly,
      	"GEN.ORT.YYY.MYY10131_LA" ));
      if ( WLa == null )
      {
        Globdata.GetStateData().SetCurrentABId( "0022020333" );
        Globdata.GetStateData().SetCurrentABName( abname );
        Globdata.GetErrorData().SetErrorEncountered( ErrorData.ErrorEncounteredErrorFound );
        Globdata.GetErrorData().SetLastStatus( ErrorData.LastStatusIefAllocationError );
      }
    }
    
    public void f_22020333_init(  )
    {
      if ( NestingLevel < 2 )
      {
        WLa.Reset();
      }
      WOa.ExpIyy1ParentPreferenceId = "00000000000000000000";
      WOa.ExpErrorIyy1ComponentSeverityCode = " ";
      WOa.ExpErrorIyy1ComponentRollbackIndicator = " ";
      WOa.ExpErrorIyy1ComponentOriginServid = 0.0;
      WOa.ExpErrorIyy1ComponentContextString = "";
      WOa.ExpErrorIyy1ComponentReturnCode = 0;
      WOa.ExpErrorIyy1ComponentReasonCode = 0;
      WOa.ExpErrorIyy1ComponentChecksum = "               ";
    }
  }// end class
  
}

